from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from doc_qa.agent.tools import (
    calculate,
    make_classify_document,
    make_list_documents,
    make_search_documents,
    make_summarize_document,
)
from doc_qa.store.base import Chunk


def _chunk(filename="doc.pdf", text="Sample text.", index=0, page=1):
    return Chunk(
        chunk_id=f"{filename}::{index:04d}",
        text=text,
        filename=filename,
        page_or_line=page,
        chunk_index=index,
    )


# ---------------------------------------------------------------------------
# search_documents
# ---------------------------------------------------------------------------


def test_search_documents_returns_formatted_results(chroma_store, fake_embedder):
    chunks = [_chunk("loan.pdf", "Interest rate is 6.75%.", 0, 1)]
    embeddings = fake_embedder.embed_documents([c.text for c in chunks])
    chroma_store.add_chunks(chunks, embeddings)

    search_fn = make_search_documents(chroma_store, fake_embedder)
    result = search_fn.invoke({"query": "interest rate", "top_k": 1})

    assert "Source: loan.pdf" in result
    assert "Score:" in result
    assert "Interest rate" in result


def test_search_documents_no_results_returns_sentinel(chroma_store, fake_embedder):
    search_fn = make_search_documents(chroma_store, fake_embedder)
    result = search_fn.invoke({"query": "something not in store", "top_k": 5})
    assert result == "No results found for query."


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


def test_list_documents_returns_table(chroma_store, fake_embedder):
    chunks = [_chunk("doc_a.pdf", "text a", 0), _chunk("doc_b.txt", "text b", 0)]
    embeddings = fake_embedder.embed_documents([c.text for c in chunks])
    chroma_store.add_chunks(chunks, embeddings)

    list_fn = make_list_documents(chroma_store)
    result = list_fn.invoke({})

    assert "doc_a.pdf" in result
    assert "doc_b.txt" in result
    assert "Total: 2" in result


def test_list_documents_empty_store(chroma_store):
    list_fn = make_list_documents(chroma_store)
    result = list_fn.invoke({})
    assert "No documents ingested yet." in result


# ---------------------------------------------------------------------------
# summarize_document
# ---------------------------------------------------------------------------


def test_summarize_document_calls_llm(chroma_store, fake_embedder):
    chunks = [_chunk("report.pdf", "Annual report content.", 0)]
    embeddings = fake_embedder.embed_documents([c.text for c in chunks])
    chroma_store.add_chunks(chunks, embeddings)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Summary of the report.")

    summarize_fn = make_summarize_document(chroma_store, mock_llm)
    result = summarize_fn.invoke({"filename": "report.pdf"})

    mock_llm.invoke.assert_called_once()
    call_arg = mock_llm.invoke.call_args[0][0]
    assert "Annual report content." in call_arg
    assert result == "Summary of the report."


def test_summarize_document_missing_file(chroma_store):
    mock_llm = MagicMock()
    summarize_fn = make_summarize_document(chroma_store, mock_llm)
    result = summarize_fn.invoke({"filename": "missing.pdf"})
    assert "No content found" in result
    mock_llm.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# classify_document
# ---------------------------------------------------------------------------


def test_classify_document_returns_known_type(chroma_store, fake_embedder):
    chunks = [_chunk("note.pdf", "I promise to pay the sum of $200,000.", 0)]
    embeddings = fake_embedder.embed_documents([c.text for c in chunks])
    chroma_store.add_chunks(chunks, embeddings)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="Promissory Note: The text contains a promise to pay."
    )

    classify_fn = make_classify_document(chroma_store, mock_llm)
    result = classify_fn.invoke({"filename": "note.pdf"})

    mock_llm.invoke.assert_called_once()
    assert "Promissory Note" in result


def test_classify_document_missing_file(chroma_store):
    mock_llm = MagicMock()
    classify_fn = make_classify_document(chroma_store, mock_llm)
    result = classify_fn.invoke({"filename": "ghost.pdf"})
    assert "No content found" in result
    mock_llm.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# calculate
# ---------------------------------------------------------------------------


def test_calculate_pmt():
    # Standard 30yr mortgage: rate=0.005 (6%/12), nper=360, pv=200000 → ~$1199.10
    result = calculate.invoke({"expression": "PMT(0.005, 360, 200000)"})
    assert "1,199" in result or "1199" in result


def test_calculate_simple_arithmetic():
    result = calculate.invoke({"expression": "1200 * 12"})
    assert "14400" in result


def test_calculate_addition():
    result = calculate.invoke({"expression": "150000 + 2500"})
    assert "152500" in result


def test_calculate_unknown_expression_returns_error_string():
    result = calculate.invoke({"expression": "AMORTIZE(nonsense here @@@)"})
    assert "Cannot compute" in result


def test_calculate_division_by_zero():
    result = calculate.invoke({"expression": "100 / 0"})
    assert "Cannot compute" in result
