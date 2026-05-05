"""Streamlit chat interface for the doc-qa agent."""

import tempfile
from pathlib import Path

import streamlit as st

from config.settings import settings
from doc_qa.agent.runner import AgentRunner
from doc_qa.embeddings import get_embedding_provider
from doc_qa.ingestion.pipeline import ingest_file
from doc_qa.store.chroma import ChromaVectorStore

st.set_page_config(page_title="Doc QA Agent", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Session state initialisation (runs once per browser session)
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "runner" not in st.session_state:
        store = ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        embedder = get_embedding_provider()
        st.session_state.store = store
        st.session_state.embedder = embedder
        st.session_state.runner = AgentRunner(store, embedder)
        st.session_state.agent_session = st.session_state.runner.new_session()
        st.session_state.messages = []  # display list: {"role": ..., "content": ...}


_init_session()

runner: AgentRunner = st.session_state.runner
agent_session = st.session_state.agent_session
store: ChromaVectorStore = st.session_state.store
embedder = st.session_state.embedder


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # --- Document Corpus ---
    st.header("📚 Document Corpus")
    filenames = store.list_documents()
    if filenames:
        for fn in filenames:
            count = len(store.get_chunks_for_document(fn))
            st.caption(f"• {fn} ({count} chunks)")
    else:
        st.caption("No documents ingested yet.")

    st.divider()

    # --- Ingest New File ---
    st.header("⬆️ Ingest New File")
    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md", "csv", "json", "docx"],
        label_visibility="collapsed",
    )
    if uploaded and st.button("Ingest", use_container_width=True):
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)
        with st.spinner(f"Ingesting {uploaded.name}…"):
            result = ingest_file(tmp_path, store, embedder)
        tmp_path.unlink(missing_ok=True)
        if result.skipped:
            st.info(f"{uploaded.name}: already ingested (skipped).")
        else:
            st.success(
                f"{uploaded.name}: {result.chunks_created} chunks added "
                f"in {result.total_time_s:.1f}s."
            )
        st.rerun()

    st.divider()

    # --- Settings ---
    st.header("⚙️ Settings")
    st.caption(f"Embedding: `{settings.embedding_provider}`")
    st.caption(f"LLM: `{settings.llm_provider}`")
    st.caption(f"Collection: `{settings.chroma_collection_name}`")

    st.divider()

    # --- MLflow ---
    st.header("📊 MLflow")
    st.markdown("[Open MLflow UI](http://localhost:5000)", unsafe_allow_html=True)
    if agent_session.mlflow_run_id:
        st.caption(f"Run ID: `{agent_session.mlflow_run_id}`")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("📄 Banking Document Q&A")

# Replay message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user input
if prompt := st.chat_input("Ask a question about your documents…"):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call agent
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking…"):
            response = runner.run_turn(agent_session, prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
