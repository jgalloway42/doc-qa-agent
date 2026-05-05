from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from doc_qa.agent.graph import AgentState, build_graph, validate_tool_results

# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


def test_graph_builds_without_error(chroma_store, fake_embedder, mock_llm):
    graph = build_graph(chroma_store, fake_embedder, mock_llm)
    assert graph is not None


def test_graph_returns_ai_message(chroma_store, fake_embedder, mock_llm):
    graph = build_graph(chroma_store, fake_embedder, mock_llm)
    initial_state = {
        "messages": [HumanMessage(content="What is in these documents?")],
        "retry_count": 0,
        "last_tool_name": "",
        "last_tool_empty": False,
    }
    result = graph.invoke(initial_state)
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1


# ---------------------------------------------------------------------------
# validate_tool_results — unit tests (no graph invocation needed)
# ---------------------------------------------------------------------------


def _make_state(
    tool_name: str,
    tool_content: str,
    retry_count: int = 0,
) -> AgentState:
    return AgentState(
        messages=[
            ToolMessage(
                content=tool_content,
                name=tool_name,
                tool_call_id="fake-id",
            )
        ],
        retry_count=retry_count,
        last_tool_name="",
        last_tool_empty=False,
    )


def test_validate_tool_results_no_op_on_good_results():
    state = _make_state("search_documents", "Interest rate is 6.75% per annum.")
    result = validate_tool_results(state)
    assert result.get("last_tool_empty") is False
    assert "retry_count" not in result or result.get("retry_count") == 0
    # No SystemMessage injected
    assert not any(isinstance(m, SystemMessage) for m in result.get("messages", []))


def test_validate_tool_results_injects_hint_on_empty_search():
    state = _make_state("search_documents", "No results found for query.")
    result = validate_tool_results(state)
    assert result["last_tool_empty"] is True
    assert result["retry_count"] == 1
    injected = result.get("messages", [])
    assert any(isinstance(m, SystemMessage) for m in injected)
    hint_text = next(m.content for m in injected if isinstance(m, SystemMessage))
    assert "broadening" in hint_text.lower() or "broader" in hint_text.lower()


def test_validate_tool_results_does_not_retry_beyond_max():
    from config.settings import settings

    state = _make_state(
        "search_documents",
        "No results found for query.",
        retry_count=settings.max_tool_retries,
    )
    result = validate_tool_results(state)
    # No hint injected when at max retries
    assert not any(isinstance(m, SystemMessage) for m in result.get("messages", []))
    assert result.get("last_tool_empty") is False


def test_validate_tool_results_ignores_non_search_tools():
    # calculate tool returning something that looks like "no results"
    state = _make_state("calculate", "No results found for query.")
    result = validate_tool_results(state)
    assert not any(isinstance(m, SystemMessage) for m in result.get("messages", []))
    assert result.get("last_tool_empty") is False
    assert result.get("last_tool_name") == "calculate"
