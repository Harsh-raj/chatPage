"""
Logic for the frontend, kept separate from main.py's top-level script body
(sidebar rendering, the readiness-gate loop, the chat loop) specifically so
it can be imported and unit-tested directly -- `import app.main` runs the
whole Streamlit script immediately (that's how Streamlit apps work, no
`if __name__ == "__main__"` guard), including a real polling loop with
`time.sleep`, which makes it unsuitable for a plain `from app.main import
some_function` in a test. Nothing in this module runs on import.
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_KEY = os.environ.get("API_KEY")


def _auth_headers() -> dict:
    """Headers for calls to the gateway/ingestion services -- the frontend
    doesn't enforce the key itself (it's not one of the protected services,
    see app/auth.py in the other services), it just needs to know the key
    so its own calls to the protected backends succeed."""
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _describe_ingest_event(event: dict) -> str | None:
    """Progress-event -> human-readable line, for the document upload log."""
    event_type = event["type"]
    if event_type == "start":
        return f"Found {event['num_pages']} page(s). Extracting text, tables, and images..."
    if event_type == "page_extracted":
        return f"Extracted page {event['page']}/{event['num_pages']} ({event['documents_so_far']} chunks so far)"
    if event_type == "summarizing_page":
        return f"Summarizing page {event['page']}/{event['num_pages']}..."
    if event_type == "summarizing_document":
        return "Building a whole-document summary..."
    if event_type == "extraction_done":
        counts = ", ".join(f"{n} {level}" for level, n in event["counts"].items()) or "nothing"
        return f"Extraction complete: {counts}."
    if event_type == "index_progress":
        if event["total"] == 0:
            return event.get("message", "Nothing to index.")
        return f"Indexed {event['indexed']}/{event['total']} chunks into retrieval"
    if event_type == "done":
        return f"Done -- indexed {event['num_documents']} chunk(s) from {event['filename']}."
    if event_type == "error":
        return f"Error: {event['message']}"
    return None  # unrecognized event type -- skip rather than print raw JSON


def ingest_pdf_with_progress(
    ingestion_url: str,
    uploaded_file,
    chunk_size: int,
    overlap: int,
    with_summaries: bool,
    with_tables: bool,
    with_ocr: bool,
) -> None:
    """
    Uploads a PDF to the ingestion service and renders its NDJSON progress
    stream as a live log inside an st.status() box, so a multi-minute OCR +
    summarization run on a large scanned PDF shows what's happening instead
    of leaving the sidebar looking frozen.
    """
    with st.status(f"Ingesting {uploaded_file.name}...", expanded=True) as status:
        try:
            response = requests.post(
                f"{ingestion_url}/ingest/stream",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                },
                headers=_auth_headers(),
                data={
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "with_summaries": with_summaries,
                    "with_tables": with_tables,
                    "with_ocr": with_ocr,
                },
                stream=True,
                timeout=900,  # OCR + per-page summarization on a large scanned PDF can take a while
            )
            response.raise_for_status()

            saw_error = False
            saw_done = False
            warnings: list[str] = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                description = _describe_ingest_event(event)
                if description:
                    st.write(description)
                if event["type"] == "error":
                    saw_error = True
                if event["type"] == "done":
                    saw_done = True
                    warnings = event.get("warnings", [])

            if saw_error:
                status.update(label=f"Failed to ingest {uploaded_file.name}", state="error")
            elif saw_done:
                status.update(label=f"Ingested {uploaded_file.name}", state="complete")
                for w in warnings:
                    st.warning(w)
            else:
                # Stream ended without a "done" or "error" event -- the
                # ingestion service crashed or was killed mid-run.
                status.update(label="Ingestion stream ended unexpectedly", state="error")

        except requests.exceptions.RequestException as e:
            st.write(f"Could not reach the ingestion service: {e}")
            status.update(label="Could not reach the ingestion service", state="error")


def check_backend_ready(gateway_url: str) -> dict | None:
    """Returns the gateway's aggregated health response, or None if the
    gateway itself isn't reachable yet."""
    try:
        response = requests.get(f"{gateway_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    cols = st.columns(len(sources))
    for i, (col, src) in enumerate(zip(cols, sources, strict=True), start=1):
        with col, st.popover(f"📄 {i}", use_container_width=True):
            st.markdown(f"**`{src['id']}`**")
            st.caption(f"score: {src['score']:.3f}")
            st.write(src["text"])


def stream_query(gateway_url: str, query_text: str, sources_holder: list, error_holder: dict):
    """
    Streams tokens from the gateway. If the backend turns out to be
    unreachable or unready mid-conversation (e.g. a container restarted
    after this session already passed the initial readiness gate),
    error_holder["backend_down"] is set to True so the caller can reset
    the readiness gate rather than leaving a stale "ready" flag that
    would keep producing failed queries indefinitely.
    """
    try:
        with requests.post(
            f"{gateway_url}/query/stream",
            json={"query": query_text, "top_k": 5},
            headers=_auth_headers(),
            stream=True,
            timeout=150,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "sources":
                    sources_holder.extend(event["sources"])
                elif event["type"] == "token":
                    yield event["text"]
                elif event["type"] == "done":
                    return
    except requests.exceptions.ConnectionError:
        error_holder["backend_down"] = True
        yield "Could not connect to the gateway service. Re-checking backend readiness..."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status and status >= 500:
            # A 5xx from the gateway usually means one of its downstream
            # services (retrieval/generation) isn't actually ready, even
            # though this session previously passed the readiness gate.
            error_holder["backend_down"] = True
            yield f"A backend service isn't ready yet (error {status}). Re-checking backend readiness..."
        else:
            yield f"Gateway returned an error: {e}"
    except requests.exceptions.ReadTimeout:
        yield (
            "The model is taking longer than expected to respond. This can happen on "
            "the first request after Ollama has been idle. Please try again."
        )
