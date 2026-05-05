"""Streamlit chat interface for the doc-qa agent."""

import tempfile
from pathlib import Path

import requests
import streamlit as st

from config.settings import settings

st.set_page_config(page_title="Doc QA Agent", page_icon="📄", layout="wide")

API = settings.api_base_url


# ---------------------------------------------------------------------------
# Session state initialisation (runs once per browser session)
# ---------------------------------------------------------------------------


def _init_session() -> None:
    if "session_id" not in st.session_state:
        resp = requests.post(f"{API}/sessions", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        st.session_state.session_id = data["session_id"]
        st.session_state.mlflow_run_id = data.get("mlflow_run_id")
        st.session_state.messages = []  # display list: {"role": ..., "content": ...}


_init_session()


def _list_documents() -> list[dict]:
    resp = requests.get(f"{API}/documents", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # --- Document Corpus ---
    st.header("📚 Document Corpus")
    docs = _list_documents()
    if docs:
        for doc in docs:
            st.caption(f"• {doc['filename']} ({doc['chunk_count']} chunks)")
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
        with st.spinner(f"Ingesting {uploaded.name}…"), open(tmp_path, "rb") as fh:
            resp = requests.post(
                f"{API}/documents/ingest",
                files={"file": (uploaded.name, fh)},
                timeout=120,
            )
        tmp_path.unlink(missing_ok=True)
        if resp.status_code == 200:
            result = resp.json()
            if result["skipped"]:
                st.info(f"{uploaded.name}: already ingested (skipped).")
            else:
                st.success(
                    f"{uploaded.name}: {result['chunks_created']} chunks added "
                    f"in {result['total_time_s']:.1f}s."
                )
        else:
            st.error(f"Ingest failed: {resp.text}")
        st.rerun()

    st.divider()

    # --- Settings ---
    st.header("⚙️ Settings")
    st.caption(f"Embedding: `{settings.embedding_provider}`")
    st.caption(f"LLM: `{settings.llm_provider}`")
    st.caption(f"Collection: `{settings.chroma_collection_name}`")
    st.caption(f"API: `{API}`")

    st.divider()

    # --- MLflow ---
    st.header("📊 MLflow")
    st.markdown("[Open MLflow UI](http://localhost:5000)", unsafe_allow_html=True)
    if st.session_state.get("mlflow_run_id"):
        st.caption(f"Run ID: `{st.session_state.mlflow_run_id}`")


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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking…"):
            resp = requests.post(
                f"{API}/sessions/{st.session_state.session_id}/chat",
                json={"message": prompt},
                timeout=120,
            )
            resp.raise_for_status()
            response = resp.json()["response"]
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
