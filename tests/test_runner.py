from langchain_core.messages import AIMessage, HumanMessage

from doc_qa.agent.runner import AgentRunner, AgentSession


def _make_runner(chroma_store, fake_embedder, mock_llm, mocker):
    """Build a runner with a mocked graph so no real LLM calls happen."""
    mocker.patch("doc_qa.agent.runner.make_llm", return_value=mock_llm)
    # Patch build_graph to return a mock graph whose invoke returns a fixed AIMessage
    mock_graph = mocker.MagicMock()
    mocker.patch("doc_qa.agent.runner.build_graph", return_value=mock_graph)
    runner = AgentRunner(chroma_store, fake_embedder)
    return runner, mock_graph


# ---------------------------------------------------------------------------
# new_session
# ---------------------------------------------------------------------------


def test_new_session_returns_session_with_id(chroma_store, fake_embedder, mock_llm, mocker):
    runner, _ = _make_runner(chroma_store, fake_embedder, mock_llm, mocker)
    session = runner.new_session()
    assert isinstance(session, AgentSession)
    assert len(session.session_id) > 0


# ---------------------------------------------------------------------------
# run_turn
# ---------------------------------------------------------------------------


def _setup_graph_response(mock_graph, response_text: str):
    """Configure mock graph to return a single AIMessage response."""
    mock_graph.invoke.return_value = {
        "messages": [AIMessage(content=response_text)],
        "retry_count": 0,
        "last_tool_name": "",
        "last_tool_empty": False,
    }


def test_run_turn_returns_string(chroma_store, fake_embedder, mock_llm, mocker):
    runner, mock_graph = _make_runner(chroma_store, fake_embedder, mock_llm, mocker)
    session = runner.new_session()
    _setup_graph_response(mock_graph, "The loan amount is $200,000 per loan_application.pdf.")
    result = runner.run_turn(session, "What is the loan amount?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_run_turn_appends_to_history(chroma_store, fake_embedder, mock_llm, mocker):
    runner, mock_graph = _make_runner(chroma_store, fake_embedder, mock_llm, mocker)
    session = runner.new_session()

    _setup_graph_response(mock_graph, "Answer referencing loan_application.pdf.")
    runner.run_turn(session, "First question?")
    _setup_graph_response(mock_graph, "Another answer about loan_application.pdf.")
    runner.run_turn(session, "Second question?")

    # 2 turns → 4 messages: HumanMessage + AIMessage per turn
    assert len(session.history) == 4
    assert isinstance(session.history[0], HumanMessage)
    assert isinstance(session.history[1], AIMessage)
    assert isinstance(session.history[2], HumanMessage)
    assert isinstance(session.history[3], AIMessage)


def test_run_turn_uses_tool_calls_for_grounding(chroma_store, fake_embedder, mock_llm, mocker):
    from langchain_core.messages import ToolMessage

    runner, mock_graph = _make_runner(chroma_store, fake_embedder, mock_llm, mocker)
    session = runner.new_session()

    # Response has no filename but search_documents was called — should be grounded
    mock_graph.invoke.return_value = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_documents", "id": "c1", "args": {"query": "rate"}}],
            ),
            ToolMessage(
                content="Rate is 6.75% per mortgage_faq.md page 3.",
                name="search_documents",
                tool_call_id="c1",
            ),
            AIMessage(content="The current mortgage rate is 6.75%."),
        ],
        "retry_count": 0,
        "last_tool_name": "search_documents",
        "last_tool_empty": False,
    }

    result = runner.run_turn(session, "What is the rate?")
    assert not result.startswith("⚠️")


# ---------------------------------------------------------------------------
# _check_grounded
# ---------------------------------------------------------------------------


def test_check_grounded_true_when_filename_present(chroma_store, fake_embedder, mock_llm, mocker):
    runner, _ = _make_runner(chroma_store, fake_embedder, mock_llm, mocker)
    # Seed the store with a document so list_documents returns something
    from doc_qa.store.base import Chunk

    chunks = [Chunk("f::0000", "text", "loan_application.pdf", 1, 0)]
    embeddings = fake_embedder.embed_documents(["text"])
    chroma_store.add_chunks(chunks, embeddings)

    assert runner._check_grounded("According to loan_application.pdf, the rate is 6.75%.") is True


def test_check_grounded_false_when_no_filename(chroma_store, fake_embedder, mock_llm, mocker):
    runner, _ = _make_runner(chroma_store, fake_embedder, mock_llm, mocker)
    from doc_qa.store.base import Chunk

    chunks = [Chunk("f::0000", "text", "loan_application.pdf", 1, 0)]
    embeddings = fake_embedder.embed_documents(["text"])
    chroma_store.add_chunks(chunks, embeddings)

    assert runner._check_grounded("The interest rate is 6.5% per current market rates.") is False


def test_check_grounded_true_on_ungrounded_prefix(chroma_store, fake_embedder, mock_llm, mocker):
    runner, _ = _make_runner(chroma_store, fake_embedder, mock_llm, mocker)
    assert runner._check_grounded("UNGROUNDED: I could not find relevant information.") is True


def test_check_grounded_false_on_empty_response(chroma_store, fake_embedder, mock_llm, mocker):
    runner, _ = _make_runner(chroma_store, fake_embedder, mock_llm, mocker)
    assert runner._check_grounded("") is False
    assert runner._check_grounded("   ") is False


# ---------------------------------------------------------------------------
# Grounding failure formatting in run_turn
# ---------------------------------------------------------------------------


def test_run_turn_formats_grounding_failure_when_ungrounded(
    chroma_store, fake_embedder, mock_llm, mocker
):
    runner, mock_graph = _make_runner(chroma_store, fake_embedder, mock_llm, mocker)
    session = runner.new_session()
    # Response with no filename → grounding failure
    _setup_graph_response(mock_graph, "The answer is 42 based on general knowledge.")
    result = runner.run_turn(session, "What is the rate?")
    assert result.startswith("⚠️")


def test_run_turn_formats_grounding_failure_on_ungrounded_prefix(
    chroma_store, fake_embedder, mock_llm, mocker
):
    runner, mock_graph = _make_runner(chroma_store, fake_embedder, mock_llm, mocker)
    session = runner.new_session()
    _setup_graph_response(mock_graph, "UNGROUNDED: I could not find information about rates.")
    result = runner.run_turn(session, "What is the rate?")
    assert result.startswith("⚠️")
    assert "I could not find" in result


def test_run_turn_grounded_response_returned_as_is(chroma_store, fake_embedder, mock_llm, mocker):
    runner, mock_graph = _make_runner(chroma_store, fake_embedder, mock_llm, mocker)
    session = runner.new_session()

    # Seed store so the filename is recognized
    from doc_qa.store.base import Chunk

    chunks = [Chunk("f::0000", "text", "mortgage_faq.md", 1, 0)]
    embeddings = fake_embedder.embed_documents(["text"])
    chroma_store.add_chunks(chunks, embeddings)

    answer = "The rate is 6.75% as stated in mortgage_faq.md page 2."
    _setup_graph_response(mock_graph, answer)
    result = runner.run_turn(session, "What is the mortgage rate?")
    assert not result.startswith("⚠️")
    assert result == answer
