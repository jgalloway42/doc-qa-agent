"""Per-session conversation manager and single run_turn() entry point."""

import time
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from doc_qa.agent.graph import build_graph, make_llm
from doc_qa.embeddings import EmbeddingProvider
from doc_qa.store.base import VectorStore

UNGROUNDED_PREFIX = "UNGROUNDED:"

_CONTENT_TOOLS = {"search_documents", "summarize_document", "classify_document", "list_documents"}

_SYSTEM_PROMPT = """\
You are a precise document analysis assistant for a banking institution.
You have access to a corpus of loan and banking documents.
Always cite your sources (document name and page number) when answering.
Use the calculate tool for any numerical computations.
Use classify_document when asked to identify document types.
Use summarize_document when asked to summarize a specific document by name.
Use search_documents to find passages relevant to a question.
If a search returns no results, try a different, broader query before concluding
the information is not available.
Reason step by step before producing a final answer.

GROUNDING RULES — these are mandatory, not suggestions:
1. You MUST only answer from information retrieved via the search_documents,
   summarize_document, classify_document, or list_documents tools.
2. You MUST NOT answer from general knowledge or training data.
3. If after retrying your search you still have no retrieved evidence, you MUST
   respond with exactly this format:
   "UNGROUNDED: I was unable to find information about [topic] in the available documents."
4. You MUST NOT fabricate citations, page numbers, or document names.
5. Every factual claim in your response must be traceable to a retrieved chunk.\
"""


@dataclass
class AgentSession:
    session_id: str
    history: list[BaseMessage] = field(default_factory=list)


class AgentRunner:
    def __init__(self, store: VectorStore, embedder: EmbeddingProvider) -> None:
        self._store = store
        self._embedder = embedder
        self._llm = make_llm()
        self._graph = build_graph(store, embedder, self._llm)

    def new_session(self) -> AgentSession:
        """Create a new session. Initialises LangSmith tracing (no-op if not configured)."""
        from doc_qa.observability import setup_langsmith

        setup_langsmith()
        return AgentSession(session_id=str(uuid.uuid4()))

    def run_turn(self, session: AgentSession, user_message: str) -> str:
        """Invoke the graph, apply grounding check, return response.

        LangSmith automatically traces the full graph execution — every node,
        tool call, and LLM response — when LANGCHAIN_TRACING_V2 is enabled.
        """
        t_start = time.monotonic()
        system_msg = SystemMessage(content=_SYSTEM_PROMPT)
        human_msg = HumanMessage(content=user_message)

        initial_state = {
            "messages": [system_msg, *session.history, human_msg],
            "retry_count": 0,
            "last_tool_name": "",
            "last_tool_empty": False,
        }
        result_state = self._graph.invoke(initial_state)
        _ = time.monotonic() - t_start  # latency available if needed for future use

        # Extract final AIMessage
        final_ai_msg: AIMessage | None = None
        for msg in reversed(list(result_state["messages"])):
            if isinstance(msg, AIMessage):
                final_ai_msg = msg
                break

        response_text = final_ai_msg.content if final_ai_msg else ""
        if not isinstance(response_text, str):
            response_text = str(response_text)

        # Collect tool call names for grounding check
        tool_call_names: list[str] = []
        for msg in result_state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_call_names.append(tc["name"])

        # Grounding check — skip filename requirement when only calculate was called,
        # since those responses derive from prior retrieved context, not a new search.
        calculate_only = bool(tool_call_names) and all(t == "calculate" for t in tool_call_names)
        grounded = calculate_only or self._check_grounded(response_text, tool_call_names)
        if not grounded:
            final_response = self._format_grounding_failure()
        elif response_text.startswith(UNGROUNDED_PREFIX):
            stripped = response_text[len(UNGROUNDED_PREFIX) :].strip()
            doc_list = "\n".join(f"• {fn}" for fn in self._store.list_documents())
            final_response = (
                f"⚠️ {stripped}\n\nDocuments available:\n{doc_list}" if doc_list else f"⚠️ {stripped}"
            )
        else:
            final_response = response_text

        # Append to session history
        session.history.append(human_msg)
        session.history.append(AIMessage(content=final_response))

        return final_response

    def _check_grounded(self, response: str, tool_call_names: list[str] | None = None) -> bool:
        """
        Return True if:
        - response starts with UNGROUNDED_PREFIX, or
        - a content-retrieval tool was called (search_documents, summarize_document, etc.), or
        - response text contains a known filename.
        Return False if non-empty but has no evidence of grounding.
        """
        if response.startswith(UNGROUNDED_PREFIX):
            return True
        if len(response.strip()) == 0:
            return False
        if tool_call_names and any(t in _CONTENT_TOOLS for t in tool_call_names):
            return True
        known_filenames = set(self._store.list_documents())
        return any(fname in response for fname in known_filenames)

    def _format_grounding_failure(self) -> str:
        """Build the warning message shown when no grounding evidence is found."""
        filenames = self._store.list_documents()
        doc_bullets = "\n".join(f"• {fn}" for fn in filenames)
        return (
            "⚠️ I was unable to find relevant information about this question\n"
            "in the current document corpus.\n\n"
            f"Documents available:\n{doc_bullets}\n\n"
            "If this topic is covered in another document, you can ingest it\n"
            "using the sidebar uploader."
        )
