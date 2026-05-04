"""LangGraph ReAct agent graph for document Q&A."""

import operator
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from doc_qa.agent.tools import (
    calculate,
    make_classify_document,
    make_list_documents,
    make_search_documents,
    make_summarize_document,
)
from doc_qa.embeddings import EmbeddingProvider
from doc_qa.store.base import VectorStore


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    retry_count: int
    last_tool_name: str
    last_tool_empty: bool


def validate_tool_results(state: AgentState) -> dict:
    """
    Inspect the most recent ToolMessage. If search_documents returned no results
    and retry_count < max_tool_retries, inject a broadening hint and increment retry_count.
    """
    from config.settings import settings

    messages = list(state["messages"])
    retry_count = state["retry_count"]

    last_tool_msg: ToolMessage | None = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            last_tool_msg = msg
            break

    if last_tool_msg is None:
        return {"last_tool_empty": False}

    tool_name = last_tool_msg.name or ""

    if (
        tool_name == "search_documents"
        and "No results found" in str(last_tool_msg.content)
        and retry_count < settings.max_tool_retries
    ):
        hint = SystemMessage(
            content=(
                "The previous search returned no results. "
                "Try broadening the query — use fewer keywords or more general terms."
            )
        )
        return {
            "messages": [hint],
            "last_tool_name": tool_name,
            "last_tool_empty": True,
            "retry_count": retry_count + 1,
        }

    return {
        "last_tool_name": tool_name,
        "last_tool_empty": False,
    }


def make_llm():
    """Construct the LLM from settings. Used by AgentRunner."""
    from config.settings import settings

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        )
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )
    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")


def build_graph(store: VectorStore, embedder: EmbeddingProvider, llm):
    """
    Build and compile a LangGraph ReAct-style agent graph.

    Nodes: agent → tools → validate_tool_results → agent (loop)
    The graph terminates when the agent produces a response with no tool calls.
    """
    tools = [
        make_search_documents(store, embedder),
        make_list_documents(store),
        make_summarize_document(store, llm),
        make_classify_document(store, llm),
        calculate,
    ]

    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(list(state["messages"]))
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("validate_tool_results", validate_tool_results)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "validate_tool_results")
    graph.add_edge("validate_tool_results", "agent")

    return graph.compile()
