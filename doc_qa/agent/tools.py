"""All five LangGraph agent tools. Tools are stateless; store/embedder/llm are injected via closure."""

import math
import re

from langchain_core.tools import tool

from doc_qa.embeddings import EmbeddingProvider
from doc_qa.store.base import VectorStore

# ---------------------------------------------------------------------------
# Tool 1: search_documents
# ---------------------------------------------------------------------------


def make_search_documents(store: VectorStore, embedder: EmbeddingProvider):
    @tool
    def search_documents(query: str, top_k: int = 5) -> str:
        """Search the document corpus for chunks relevant to a query.
        Returns formatted list of matching passages with source citations."""
        query_embedding = embedder.embed_query(query)
        results = store.search(query_embedding, top_k=top_k)
        if not results:
            return "No results found for query."
        lines = []
        for i, r in enumerate(results, 1):
            c = r.chunk
            lines.append(
                f"[{i}] Source: {c.filename} (page {c.page_or_line}, chunk {c.chunk_index:03d})\n"
                f"Score: {r.score:.2f}\n"
                f"Text: {c.text}"
            )
        return "\n\n".join(lines)

    return search_documents


# ---------------------------------------------------------------------------
# Tool 2: list_documents
# ---------------------------------------------------------------------------


def make_list_documents(store: VectorStore):
    @tool
    def list_documents() -> str:
        """List all documents currently ingested in the knowledge base.
        Returns filenames and chunk counts."""
        filenames = store.list_documents()
        if not filenames:
            return "No documents ingested yet."
        header = f"{'Filename':<50} {'Chunks':>6}"
        sep = "-" * 58
        rows = []
        for fn in filenames:
            count = len(store.get_chunks_for_document(fn))
            rows.append(f"{fn:<50} {count:>6}")
        return "\n".join([header, sep] + rows + [f"\nTotal: {len(filenames)} document(s)"])

    return list_documents


# ---------------------------------------------------------------------------
# Tool 3: summarize_document
# ---------------------------------------------------------------------------


def make_summarize_document(store: VectorStore, llm):
    @tool
    def summarize_document(filename: str) -> str:
        """Summarize the full content of a specific document.
        Uses all chunks from the document to produce a concise summary."""
        chunks = store.get_chunks_for_document(filename)
        if not chunks:
            return f"No content found for document: {filename}"
        full_text = " ".join(c.text for c in chunks)
        if len(full_text) > 8000:
            full_text = full_text[:8000]
        prompt = (
            f"Summarize the following banking document '{filename}' concisely in 3-5 sentences:\n\n"
            f"{full_text}"
        )
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    return summarize_document


# ---------------------------------------------------------------------------
# Tool 4: classify_document
# ---------------------------------------------------------------------------

_KNOWN_TYPES = (
    "Promissory Note",
    "Deed of Trust",
    "HUD-1 Settlement Statement",
    "TILA Disclosure",
    "Uniform Residential Loan Application (1003)",
    "Appraisal Report",
    "Closing Disclosure",
    "Loan Estimate",
    "Rate Sheet / Pricing Schedule",
    "Underwriting Guidelines",
    "FAQ / Product Catalog",
)


def make_classify_document(store: VectorStore, llm):
    @tool
    def classify_document(filename: str) -> str:
        """Classify a banking document into its standard form type.
        Identifies document type (e.g., Promissory Note, Deed of Trust,
        HUD-1 Settlement Statement, TILA Disclosure, Uniform Residential
        Loan Application (1003), Appraisal Report, Closing Disclosure).
        Returns the classification with a confidence rationale."""
        chunks = store.get_chunks_for_document(filename)
        if not chunks:
            return f"No content found for document: {filename}"
        preview = " ".join(c.text for c in chunks[:3])
        types_list = "\n".join(f"- {t}" for t in _KNOWN_TYPES)
        prompt = (
            f"Classify the following banking document excerpt from '{filename}' into one of these types:\n"
            f"{types_list}\n\n"
            f"Document excerpt:\n{preview}\n\n"
            "Respond with: <Type>: <one-sentence rationale>"
        )
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    return classify_document


# ---------------------------------------------------------------------------
# Tool 5: calculate
# ---------------------------------------------------------------------------


def _pmt(rate: float, nper: int, pv: float) -> float:
    """Standard annuity payment formula."""
    if rate == 0:
        return pv / nper
    return pv * rate * (1 + rate) ** nper / ((1 + rate) ** nper - 1)


def _parse_pmt(expr: str) -> str | None:
    """Try to parse PMT(rate, nper, pv) call. Returns formatted result or None."""
    m = re.fullmatch(r"PMT\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)", expr.strip(), re.IGNORECASE)
    if not m:
        return None
    try:
        rate = float(m.group(1))
        nper = int(float(m.group(2)))
        pv = float(m.group(3))
    except ValueError:
        return None
    result = _pmt(rate, nper, pv)
    return f"Monthly payment: ${result:,.2f}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a financial math expression safely.
    Supports: basic arithmetic, loan payment formula PMT(rate, nper, pv),
    and simple numeric expressions.
    Input: a natural-language or formula string.
    Returns: computed result with units."""
    expr = expression.strip()

    # PMT function
    pmt_result = _parse_pmt(expr)
    if pmt_result is not None:
        return pmt_result

    # Safe arithmetic via simpleeval
    try:
        from simpleeval import EvalWithCompoundTypes, InvalidExpression

        evaluator = EvalWithCompoundTypes(
            functions={"sqrt": math.sqrt, "abs": abs, "round": round, "pow": pow},
        )
        result = evaluator.eval(expr)
        return str(result)
    except (InvalidExpression, ValueError, ZeroDivisionError, SyntaxError) as exc:
        return f"Cannot compute: {exc}"
    except Exception as exc:
        return f"Cannot compute: {exc}"
