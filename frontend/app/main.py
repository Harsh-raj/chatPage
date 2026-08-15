import os
import time
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logic import (
    check_backend_ready, render_sources, stream_query, ingest_pdf_with_progress,
)

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
INGESTION_URL = os.environ.get("INGESTION_URL", "http://localhost:8003")

st.title("Retrieval Augmented Generation")


with st.sidebar:
    st.header("Add a document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    with st.expander("Ingestion options"):
        with_summaries = st.checkbox(
            "Generate page + document summaries", value=True,
            help="Calls the LLM once per page plus once for the whole document -- "
                 "the slowest part of ingestion, but improves answers to broad "
                 "'what is this document about' questions.",
        )
        with_tables = st.checkbox("Extract tables", value=True)
        with_ocr = st.checkbox(
            "OCR scanned pages and images", value=True,
            help="Needed for scanned PDFs with no real text layer, and for text "
                 "embedded in diagrams/figures. Skip this for speed if your PDF "
                 "is already text-based.",
        )
        chunk_size = st.number_input("Chunk size (characters)", min_value=100, max_value=4000, value=800, step=100)
        overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=2000, value=100, step=50)

    if st.button("Ingest document", disabled=uploaded_file is None, use_container_width=True):
        ingest_pdf_with_progress(
            INGESTION_URL, uploaded_file, chunk_size, overlap, with_summaries, with_tables, with_ocr,
        )


# --- Readiness gate: block the chat UI until every backend service is up ---
if "backend_ready" not in st.session_state:
    st.session_state.backend_ready = False

if not st.session_state.backend_ready:
    status_placeholder = st.empty()
    # Retrieval loads TWO models on startup (the embedder and, if enabled,
    # the reranker) -- on CPU-only hardware this has been observed to take
    # 3+ minutes on its own, so the wait budget needs real headroom above that.
    # Overridable so tests can shrink the wait/poll loop to run near-instantly
    # instead of the real multi-minute budget -- defaults are unchanged for
    # normal (non-test) runs.
    max_wait_seconds = int(os.environ.get("FRONTEND_READY_MAX_WAIT_SECONDS", "360"))
    poll_interval = int(os.environ.get("FRONTEND_READY_POLL_INTERVAL_SECONDS", "3"))
    waited = 0

    while waited < max_wait_seconds:
        health = check_backend_ready(GATEWAY_URL)

        if health and health.get("status") == "ok":
            st.session_state.backend_ready = True
            status_placeholder.empty()
            st.rerun()

        with status_placeholder.container():
            minutes, seconds = divmod(waited, 60)
            st.info(
                f"Starting up backend services... ({minutes}m {seconds}s elapsed)\n\n"
                f"This can take several minutes the first time, while the retrieval "
                f"service loads its embedding model and reranker model."
            )
            if health and "services" in health:
                for name, info in health["services"].items():
                    icon = "done" if info.get("ready") else "waiting"
                    st.write(
                        f"[{icon}] {name}: {'ready' if info.get('ready') else 'starting...'}")
            else:
                st.write("Waiting for gateway to respond...")

        time.sleep(poll_interval)
        waited += poll_interval

    st.error(
        f"Backend services haven't started after {max_wait_seconds // 60} minutes. "
        f"This is longer than expected -- check docker compose logs retrieval for errors "
        f"(e.g. an out-of-memory kill), or click below to keep waiting."
    )
    if st.button("Check again"):
        st.rerun()
    st.stop()

# --- Main chat UI, only reached once backend_ready is True ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

if prompt := st.chat_input("Provide your query here!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        sources: list = []
        error_holder = {"backend_down": False}
        full_response = st.write_stream(
            stream_query(GATEWAY_URL, prompt, sources, error_holder))
        render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })

    # If the backend turned out to be unreachable mid-conversation, drop the
    # stale "ready" flag so the readiness gate re-appears and genuinely
    # re-checks, instead of silently repeating this failure on every
    # subsequent query.
    if error_holder["backend_down"]:
        st.session_state.backend_ready = False
        st.rerun()